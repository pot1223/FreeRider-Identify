# -------------------------- R shiny를 통한 dashboard 생성 ---------------------------- # 

install.packages("shiny")
library("shiny")
install.packages("ggplot2")
library("ggplot2")
install.packages("RColorBrewer")
library("RColorBrewer") 
install.packages("dplyr")
library("dplyr")
install.packages("thematic")
library("thematic")

resultdf = read.csv("DATA/resultKakaoTalkChats.csv")
daydf  = read.csv("DATA/rawresult.csv")
daydf = data.frame(daydf)

ui <- fluidPage( # 뷰 영역으로 사용자 인터페이스를 정의한다 
  theme =bslib::bs_theme(bootswatch = "darkly"),
  titlePanel("카카오톡 톡방 대시보드"),
  titlePanel(h4(paste0("분석기간",": ", daydf[1,2]," 에서 ", daydf[nrow(daydf),2]," 까지"))),
  sidebarLayout(
    sidebarPanel(
      selectInput("이름",label="톡방인원", resultdf$이름),
      verbatimTextOutput(outputId = "myTable" , placeholder = TRUE) 
    ),
    mainPanel(
      plotOutput("plot"),
    ),
  ),
  titlePanel(plotOutput("fullplot"))
)

server <- function(input, output, session){  # 앱의 동작을 지정한다 
  thematic::thematic_shiny()
  output$myTable <- renderPrint({
    summary(subset(resultdf, resultdf$이름 == input$이름) %>% dplyr::select(속도성점수, 활동성점수, 적극성점수, 활발성점수, 연관성점수, 총점수))
  })
  output$plot <- renderPlot({
    ggplot(data=subset(resultdf, resultdf$이름 == input$이름),
           aes(x= 주차,
               y= 총점수,
               color=주차))+
      geom_line(size=1)+
      labs(title= "WAU(유저의 주차별 톡방 활동성) 점수",
           subtitle = "가중치를 준 활동성, 연관성, 적극성, 활발성, 속도성 점수의 총합",
           x ="주차",
           y= "총점수",
           ) + 
      theme(plot.title=element_text(size=25, # 제목의 크기 지정 
                                    hjust=0.5), # 제목 위치를 가운데로
            axis.title=element_text(size=10,
                                    face="bold"), # 축 제목을 굵게 
            ) +
      scale_y_continuous(breaks = seq(1,100,10)) +
      scale_x_continuous(breaks = seq(min(resultdf$주차),max(resultdf$주차)  , by = 1))  +
      geom_text(aes(y= 총점수+3, label = 총점수))+
      scale_color_gradient(low="#FF0000",
                           high="#0000FF")
  })
  
  output$fullplot <- renderPlot({
    ggplot(data=resultdf,
           aes(x= 주차,
               y= 총점수,
               color=이름))+
      geom_line(size=2)+
      geom_point() + 
      labs(title= "전체 인원 WAU",
           subtitle = "가중치를 준 활동성, 연관성, 적극성, 활발성, 속도성 점수의 총합",
           x ="주차",
           y= "총점수",
      ) + 
      theme(plot.title=element_text(size=25, # 제목의 크기 지정 
                                    hjust=0.5), # 제목 위치를 가운데로
            axis.title=element_text(size=10,
                                    face="bold"), # 축 제목을 굵게 
      ) +
      scale_y_continuous(breaks = seq(1,100,10)) +
      scale_x_continuous(breaks = seq(min(resultdf$주차),max(resultdf$주차)  , by = 1)) +
      scale_color_brewer(palette = "Paired")
    
  })
  
}

shinyApp(ui,server) # Shiny 응용 프로그램을 구성하고 실행된다
